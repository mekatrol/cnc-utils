import path from 'node:path';

import { ESLintUtils, TSESTree } from '@typescript-eslint/utils';

const PASCAL_CASE = /^[A-Z][A-Za-z0-9]*$/;
const CAMEL_CASE = /^[a-z][A-Za-z0-9]*$/;

const COMPOUND_SUFFIXES: readonly string[] = [
  '.config',
  '.spec',
  '.test',
  '.stories',
  '.story',
  '.mock',
  '.d',
  '.types',
  '.generated'
];

const DEFAULT_PASCAL_CASE_DIRECTORIES: readonly string[] = [
  'components',
  'views',
  'pages',
  'layouts',
  'classes'
];

const TEST_DIRECTORIES: readonly string[] = ['__tests__', 'tests'];

const DEFAULT_IGNORED_FILENAMES: readonly string[] = [
  'index',
  'main',
  'env',
  'vite.config',
  'eslint.config'
];

type Options = [
  {
    pascalCaseDirectories?: string[];
    ignoredFilenames?: string[];
  }?
];

type MessageIds =
  | 'expectedPascalCase'
  | 'expectedCamelCase'
  | 'kebabCaseForbidden'
  | 'appPrefixRequired';

type RuntimeExport = {
  type: 'class' | 'value';
};

const createRule = ESLintUtils.RuleCreator((): string => '');

const stripCompoundSuffixes = (filename: string): string => {
  let result = filename;
  let changed = true;

  while (changed) {
    changed = false;

    for (const suffix of COMPOUND_SUFFIXES) {
      if (result.endsWith(suffix)) {
        result = result.slice(0, -suffix.length);
        changed = true;
      }
    }
  }

  return result;
};

/**
 * Returns the names of all classes declared at the module's top level.
 *
 * This allows the rule to resolve patterns such as:
 *
 * class CustomerService {}
 * export default CustomerService;
 */
const getTopLevelClassNames = (program: TSESTree.Program): Set<string> => {
  const classNames = new Set<string>();

  for (const node of program.body) {
    if (node.type === TSESTree.AST_NODE_TYPES.ClassDeclaration && node.id) {
      classNames.add(node.id.name);
      continue;
    }

    if (
      node.type === TSESTree.AST_NODE_TYPES.ExportNamedDeclaration &&
      node.declaration?.type === TSESTree.AST_NODE_TYPES.ClassDeclaration &&
      node.declaration.id
    ) {
      classNames.add(node.declaration.id.name);
      continue;
    }

    if (
      node.type === TSESTree.AST_NODE_TYPES.ExportDefaultDeclaration &&
      node.declaration.type === TSESTree.AST_NODE_TYPES.ClassDeclaration &&
      node.declaration.id
    ) {
      classNames.add(node.declaration.id.name);
    }
  }

  return classNames;
};

/**
 * Determines whether the module's default export is a class.
 *
 * Supported examples:
 *
 * export default class CustomerService {}
 *
 * class CustomerService {}
 * export default CustomerService;
 *
 * class CustomerService {}
 * export { CustomerService as default };
 */
const hasDefaultClassExport = (
  program: TSESTree.Program,
  classNames: ReadonlySet<string>
): boolean => {
  return program.body.some((node): boolean => {
    if (node.type === TSESTree.AST_NODE_TYPES.ExportDefaultDeclaration) {
      if (
        node.declaration.type === TSESTree.AST_NODE_TYPES.ClassDeclaration ||
        node.declaration.type === TSESTree.AST_NODE_TYPES.ClassExpression
      ) {
        return true;
      }

      if (
        node.declaration.type === TSESTree.AST_NODE_TYPES.Identifier &&
        classNames.has(node.declaration.name)
      ) {
        return true;
      }
    }

    if (node.type !== TSESTree.AST_NODE_TYPES.ExportNamedDeclaration) {
      return false;
    }

    return node.specifiers.some((specifier): boolean => {
      const exportedName =
        specifier.exported.type === TSESTree.AST_NODE_TYPES.Identifier
          ? specifier.exported.name
          : specifier.exported.value;

      return (
        exportedName === 'default' &&
        specifier.local.type === TSESTree.AST_NODE_TYPES.Identifier &&
        classNames.has(specifier.local.name)
      );
    });
  });
};

/**
 * Collects runtime exports from the module.
 *
 * Type-only exports such as interfaces and type aliases are excluded because
 * they do not represent the module's runtime implementation.
 */
const getRuntimeExports = (
  program: TSESTree.Program,
  classNames: ReadonlySet<string>
): RuntimeExport[] => {
  const runtimeExports: RuntimeExport[] = [];

  for (const node of program.body) {
    if (node.type === TSESTree.AST_NODE_TYPES.ExportDefaultDeclaration) {
      if (
        node.declaration.type === TSESTree.AST_NODE_TYPES.ClassDeclaration ||
        node.declaration.type === TSESTree.AST_NODE_TYPES.ClassExpression
      ) {
        runtimeExports.push({
          type: 'class'
        });

        continue;
      }

      if (
        node.declaration.type === TSESTree.AST_NODE_TYPES.Identifier &&
        classNames.has(node.declaration.name)
      ) {
        runtimeExports.push({
          type: 'class'
        });

        continue;
      }

      runtimeExports.push({
        type: 'value'
      });

      continue;
    }

    if (node.type !== TSESTree.AST_NODE_TYPES.ExportNamedDeclaration) {
      continue;
    }

    if (node.exportKind === 'type') {
      continue;
    }

    const { declaration } = node;

    if (declaration) {
      if (
        declaration.type === TSESTree.AST_NODE_TYPES.TSInterfaceDeclaration ||
        declaration.type === TSESTree.AST_NODE_TYPES.TSTypeAliasDeclaration ||
        declaration.type === TSESTree.AST_NODE_TYPES.TSDeclareFunction
      ) {
        continue;
      }

      if (declaration.type === TSESTree.AST_NODE_TYPES.ClassDeclaration) {
        runtimeExports.push({
          type: 'class'
        });

        continue;
      }

      if (declaration.type === TSESTree.AST_NODE_TYPES.VariableDeclaration) {
        for (const variable of declaration.declarations) {
          if (variable.id.type === TSESTree.AST_NODE_TYPES.Identifier) {
            runtimeExports.push({
              type: 'value'
            });
          }
        }

        continue;
      }

      if (
        declaration.type === TSESTree.AST_NODE_TYPES.FunctionDeclaration ||
        declaration.type === TSESTree.AST_NODE_TYPES.TSEnumDeclaration
      ) {
        runtimeExports.push({
          type: 'value'
        });

        continue;
      }
    }

    for (const specifier of node.specifiers) {
      if (specifier.exportKind === 'type') {
        continue;
      }

      if (
        specifier.local.type === TSESTree.AST_NODE_TYPES.Identifier &&
        classNames.has(specifier.local.name)
      ) {
        runtimeExports.push({
          type: 'class'
        });
      } else {
        runtimeExports.push({
          type: 'value'
        });
      }
    }
  }

  return runtimeExports;
};

/**
 * A class is considered the module's primary export when:
 *
 * 1. The default export is a class; or
 * 2. The module has exactly one runtime export and that export is a class.
 *
 * Internal implementation classes do not affect filename casing.
 */
const hasPrimaryClassExport = (program: TSESTree.Program): boolean => {
  const classNames = getTopLevelClassNames(program);

  if (hasDefaultClassExport(program, classNames)) {
    return true;
  }

  const runtimeExports = getRuntimeExports(program, classNames);

  return runtimeExports.length === 1 && runtimeExports[0]?.type === 'class';
};

const isInsideDirectory = (filename: string, directoryNames: readonly string[]): boolean => {
  const directoryParts = path
    .dirname(filename)
    .split(/[\\/]/)
    .map((part: string): string => part.toLowerCase());

  return directoryNames.some((directoryName: string): boolean =>
    directoryParts.includes(directoryName.toLowerCase())
  );
};

const requireFilenameCase = createRule<Options, MessageIds>({
  name: 'requireFilenameCase',

  meta: {
    type: 'problem',

    docs: {
      description:
        'Enforce PascalCase for components and primary class modules, and camelCase for other JavaScript and TypeScript files'
    },

    schema: [
      {
        type: 'object',
        properties: {
          pascalCaseDirectories: {
            type: 'array',
            items: {
              type: 'string'
            },
            uniqueItems: true
          },

          ignoredFilenames: {
            type: 'array',
            items: {
              type: 'string'
            },
            uniqueItems: true
          }
        },
        additionalProperties: false
      }
    ],

    messages: {
      expectedPascalCase:
        'Filename "{{filename}}" must use PascalCase because it is a component or primary class module. Expected a name such as "{{example}}".',

      expectedCamelCase:
        'Filename "{{filename}}" must use camelCase. Expected a name such as "{{example}}".',

      kebabCaseForbidden:
        'Filename "{{filename}}" uses kebab-case. Kebab-case filenames are not allowed.',

      appPrefixRequired:
        'Vue component filename "{{filename}}" must start with "App". Expected a name such as "{{example}}".'
    }
  },

  defaultOptions: [{}],

  create(context, [options]) {
    const pascalCaseDirectories: readonly string[] =
      options?.pascalCaseDirectories ?? DEFAULT_PASCAL_CASE_DIRECTORIES;

    const ignoredFilenames = new Set<string>(
      options?.ignoredFilenames ?? DEFAULT_IGNORED_FILENAMES
    );

    return {
      Program(node: TSESTree.Program): void {
        const absoluteFilename = context.filename;

        // ESLint uses these values for code supplied through stdin or processors.
        if (!absoluteFilename || absoluteFilename === '<input>' || absoluteFilename === '<text>') {
          return;
        }

        const extension = path.extname(absoluteFilename).toLowerCase();
        const filenameWithExtension = path.basename(absoluteFilename);
        const filenameWithoutExtension = path.basename(absoluteFilename, extension);

        const normalizedFilename = stripCompoundSuffixes(filenameWithoutExtension);

        if (
          ignoredFilenames.has(filenameWithoutExtension) ||
          ignoredFilenames.has(normalizedFilename)
        ) {
          return;
        }

        if (extension === '.vue' && !normalizedFilename.startsWith('App')) {
          context.report({
            node,
            messageId: 'appPrefixRequired',
            data: {
              filename: filenameWithExtension,
              example: `App${normalizedFilename}`
            }
          });

          return;
        }

        /*
         * Vue files are treated as components.
         *
         * JavaScript and TypeScript files require PascalCase when:
         *   1. their primary export is a class; or
         *   2. they are inside a configured PascalCase directory.
         *
         * Internal helper classes do not affect the filename requirement.
         */
        const requiresPascalCase =
          extension === '.vue' ||
          hasPrimaryClassExport(node) ||
          (isInsideDirectory(absoluteFilename, pascalCaseDirectories) &&
            !isInsideDirectory(absoluteFilename, TEST_DIRECTORIES));

        if (normalizedFilename.includes('-')) {
          context.report({
            node,
            messageId: 'kebabCaseForbidden',
            data: {
              filename: filenameWithExtension
            }
          });

          return;
        }

        if (requiresPascalCase && !PASCAL_CASE.test(normalizedFilename)) {
          context.report({
            node,
            messageId: 'expectedPascalCase',
            data: {
              filename: filenameWithExtension,
              example: 'CustomerDetails'
            }
          });

          return;
        }

        if (!requiresPascalCase && !CAMEL_CASE.test(normalizedFilename)) {
          context.report({
            node,
            messageId: 'expectedCamelCase',
            data: {
              filename: filenameWithExtension,
              example: 'customerDetails'
            }
          });
        }
      }
    };
  }
});

export default requireFilenameCase;
