import type { Rule } from 'eslint';
import type { AST as VueAST } from 'vue-eslint-parser';

type VueParserServices = {
  defineTemplateBodyVisitor?: (
    templateVisitor: Rule.RuleListener,
    scriptVisitor?: Rule.RuleListener
  ) => Rule.RuleListener;
};

type RuleContext = Rule.RuleContext & {
  sourceCode: Rule.RuleContext['sourceCode'] & {
    parserServices: VueParserServices;
  };
};

// The app component types we do not want automation tags enforced when linting
const EXCLUDED_COMPONENTS = new Set<string>([
]);

// Must:
// - start with a lowercase letter
// - contain only lowercase letters, numbers and hyphens
// - be kebab-case (no leading/trailing/consecutive hyphens)
const AUTOMATION_PATTERN = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;

const requireAutomationProp: Rule.RuleModule = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Require valid automation prop on UI app components'
    },
    schema: [],
    messages: {
      missingAutomation: '{{name}} requires an automation prop, e.g. automation="search-input"',
      invalidAutomation:
        'automation value "{{value}}" must be lowercase kebab-case, start with a letter, and contain only letters, numbers, and hyphens'
    }
  },

  create(context): Rule.RuleListener {
    const typedContext = context as RuleContext;
    const parserServices = typedContext.sourceCode.parserServices;
    const componentFileMatch = context.filename.match(
      /[/\\]src[/\\]components[/\\]((?:App|Base)[^/\\]+)\.vue$/
    );
    const componentName = componentFileMatch?.[1];

    if (!parserServices?.defineTemplateBodyVisitor) {
      return {};
    }

    return parserServices.defineTemplateBodyVisitor({
      VElement(node: VueAST.VElement) {
        const name = node.rawName;
        const isAppComponentRoot =
          componentName !== undefined &&
          node.parent.type === 'VElement' &&
          node.parent.rawName === 'template' &&
          node.parent.parent.type === 'VDocumentFragment' &&
          node.parent.children.find((child) => child.type === 'VElement') === node;

        if (EXCLUDED_COMPONENTS.has(name)) {
          return;
        }

        const automationAttr = node.startTag.attributes.find((attr): attr is VueAST.VAttribute => {
          if (attr.type !== 'VAttribute') {
            return false;
          }

          if (!attr.directive) {
            return attr.key.name === 'automation';
          }

          const expression = attr.value?.expression;

          return (
            attr.key.name.name === 'bind' &&
            ((attr.key.argument?.type === 'VIdentifier' &&
              attr.key.argument.name === 'automation') ||
              (attr.key.argument === null &&
                expression?.type === 'CallExpression' &&
                expression.callee.type === 'Identifier' &&
                expression.callee.name === 'automation'))
          );
        });

        if (!automationAttr) {
          const requiresAutomation =
            name.startsWith('App') || name.startsWith('Base') || isAppComponentRoot;

          if (requiresAutomation) {
            typedContext.report({
              node: node.startTag,
              messageId: 'missingAutomation',
              data: { name: isAppComponentRoot ? componentName : name }
            });
          }

          return;
        }

        // Only validate static string attributes.
        // Ignore bindings such as :automation="automationId".
        if (automationAttr.value && automationAttr.value.type === 'VLiteral') {
          const value = String(automationAttr.value.value);

          if (!AUTOMATION_PATTERN.test(value)) {
            typedContext.report({
              node: automationAttr,
              messageId: 'invalidAutomation',
              data: { value }
            });
          }
        }
      }
    });
  }
};

export default requireAutomationProp;
